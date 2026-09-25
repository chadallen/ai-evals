"""HTTP forward proxy with public-address-only egress, outside the browser sandbox."""

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit


async def public_connection(host: str, port: int):
    """Resolve once, reject mixed/private answers, and connect to a checked numeric address."""
    if port not in {80, 443}:
        raise ValueError("Only web ports are permitted")
    addresses = await asyncio.get_running_loop().getaddrinfo(
        host, port, family=socket.AF_INET, type=socket.SOCK_STREAM
    )
    if not addresses:
        raise ValueError("No destination addresses")
    for _, _, _, _, address in addresses:
        ip = ipaddress.ip_address(address[0])
        if (
            ip.version != 4
            or not ip.is_global
            or ip.is_multicast
            or ip in ipaddress.ip_network("192.0.0.0/24")
            or ip in ipaddress.ip_network("192.88.99.0/24")
        ):
            raise ValueError("Non-public destination denied")
    # Do not resolve the hostname again after checking it (DNS rebinding).
    last_error = None
    for family, kind, protocol, _, address in addresses:
        sock = socket.socket(family, kind, protocol)
        sock.setblocking(False)
        try:
            await asyncio.wait_for(asyncio.get_running_loop().sock_connect(sock, address), 10)
            return await asyncio.open_connection(sock=sock)
        except OSError as exc:
            sock.close()
            last_error = exc
    raise last_error or OSError("Connection failed")


async def relay(reader, writer):
    while data := await asyncio.wait_for(reader.read(65536), 60):
        writer.write(data)
        await writer.drain()


async def handle(reader, writer):
    upstream = None
    pumps = []
    try:
        header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 15)
        lines = header.decode("iso-8859-1").split("\r\n")
        method, target, version = lines[0].split(" ")
        if version not in {"HTTP/1.0", "HTTP/1.1"}:
            raise ValueError("Unsupported HTTP version")
        url = urlsplit("//" + target if method == "CONNECT" else target)
        if url.username or url.password or not url.hostname:
            raise ValueError("Invalid destination")
        if method == "CONNECT":
            if url.port != 443 or url.path or url.query or url.fragment:
                raise ValueError("CONNECT requires port 443")
            port = 443
        else:
            if url.scheme != "http" or url.fragment:
                raise ValueError("Expected an HTTP URL")
            port = url.port or 80
            if port != 80:
                raise ValueError("HTTP requires port 80")
        remote, upstream = await public_connection(url.hostname, port)
        if method == "CONNECT":
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
        else:
            path = (url.path or "/") + ("?" + url.query if url.query else "")
            forwarded = [f"{method} {path} {version}", f"Host: {url.netloc}"]
            for line in lines[1:]:
                if not line:
                    continue
                if ":" not in line or line[0].isspace():
                    raise ValueError("Invalid header")
                name = line.split(":", 1)[0].lower()
                if name not in {"host", "connection", "proxy-connection", "proxy-authorization"}:
                    forwarded.append(line)
            forwarded.extend(["Connection: close", "", ""])
            upstream.write("\r\n".join(forwarded).encode("iso-8859-1"))
            await upstream.drain()
        pumps = [
            asyncio.create_task(relay(reader, upstream)),
            asyncio.create_task(relay(remote, writer)),
        ]
        await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
    except (
        ValueError,
        OSError,
        TimeoutError,
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
    ):
        if upstream is None:
            writer.write(
                b"HTTP/1.1 403 Forbidden\r\nContent-Length: 9\r\nConnection: close\r\n\r\nForbidden"
            )
    finally:
        for pump in pumps:
            pump.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        if upstream:
            upstream.close()
        writer.close()
        await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle, "0.0.0.0", 3128, limit=32768)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
