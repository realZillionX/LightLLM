import rpyc
import uuid
import inspect
import setproctitle
from typing import Union, Optional
from lightllm.server.core.objs import StartArgs
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.server.embed_cache.impl.naive_memory_cache import InMemoryCache
from rpyc.utils.classic import obtain
from lightllm.utils.envs_utils import get_unique_server_name


class CacheServer(rpyc.Service):
    def __init__(self, manager_impl: InMemoryCache) -> None:
        super().__init__()
        self._impl = manager_impl

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_alloc(self, md5sum_list: list[str], token_num_list: list[int]) -> Optional[list[dict]]:
        md5sum_list = obtain(md5sum_list)
        token_num_list = obtain(token_num_list)
        record = self._impl.alloc(md5sum_list, token_num_list)
        return record

    def exposed_release(self, ids: list[int]) -> None:
        ids = obtain(ids)
        return self._impl.release(ids)

    def exposed_set_items_data(self, ids: list[int]) -> None:
        ids = obtain(ids)
        return self._impl.set_items_data(ids)

    def exposed_get_items_data(self, ids: list[int]) -> list[bool]:
        ids = obtain(ids)
        return self._impl.get_items_data(ids)

    def exposed_set_items_embed(self, ids: list[int]) -> None:
        ids = obtain(ids)
        return self._impl.set_items_embed(ids)

    def exposed_get_items_embed(self, ids: list[int]) -> list[bool]:
        ids = obtain(ids)
        return self._impl.get_items_embed(ids)


def _serve_after_listening(server, pipe_writer) -> None:
    """Signal readiness only after the RPyC listener is actually bound."""

    server._listen()
    server._register()
    pipe_writer.send("init ok")
    try:
        while server.active:
            server.accept()
    except EOFError:
        pass
    finally:
        server.close()


def start_cache_manager(args: StartArgs, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::cache_manager")
    manager = InMemoryCache(args)
    service = CacheServer(manager)
    from rpyc.utils.server import ThreadedServer
    import lightllm.utils.rpyc_fix_utils as _

    t = ThreadedServer(service, port=args.cache_port, protocol_config={"allow_pickle": True})
    _serve_after_listening(t, pipe_writer)


if __name__ == "__main__":
    start_cache_manager(2233)
