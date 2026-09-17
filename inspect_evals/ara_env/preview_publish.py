"""Forward host-loopback preview traffic to the isolated generated site."""

import asyncio


async def copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def handle(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection("default", 8000)
    except OSError:
        writer.close()
        await writer.wait_closed()
        return
    await asyncio.gather(
        copy(reader, upstream_writer),
        copy(upstream_reader, writer),
        return_exceptions=True,
    )


async def main() -> None:
    server = await asyncio.start_server(handle, "0.0.0.0", 8000)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
