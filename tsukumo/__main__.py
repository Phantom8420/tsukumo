"""python -m tsukumo [serve|seed|cert|reset|eval]"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import shutil

from . import config


def make_cert(ip: str) -> tuple[str, str]:
    """Self-signed cert so phones on the LAN get a secure context (needed for sensors and mic)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    config.CERT_DIR.mkdir(exist_ok=True)
    key_path, cert_path = config.CERT_DIR / "key.pem", config.CERT_DIR / "cert.pem"
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "tsukumo.local")])
    sans = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            x509.IPAddress(ipaddress.ip_address(ip))]
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .sign(key, hashes.SHA256()))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return str(key_path), str(cert_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tsukumo")
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="run the server (default)")
    serve.add_argument("--port", type=int, default=8443)
    serve.add_argument("--http", action="store_true", help="plain HTTP (phones will refuse sensors/mic)")
    sub.add_parser("seed", help="load the demo workshop if the database is empty")
    sub.add_parser("reset", help="delete all data and reseed the demo workshop")
    ev = sub.add_parser("eval", help="measure how often surprise detection finds planted habits")
    ev.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    cmd = args.cmd or "serve"

    if cmd == "eval":
        from .evaluate import report
        print(report(range(args.seeds)))
        return

    if cmd in ("seed", "reset"):
        from .seed import seed
        from .services import Tsukumo
        from .store import Store

        if cmd == "reset" and config.DATA_DIR.exists():
            shutil.rmtree(config.DATA_DIR)
        store = Store(config.DB_PATH, config.FRAMES_DIR)
        print(seed(Tsukumo(store)))
        store.close()
        return

    import uvicorn

    from .app import create_app, lan_ip

    port = getattr(args, "port", 8443)
    use_https = not getattr(args, "http", False)
    ip = lan_ip()
    ssl = {}
    if use_https:
        key, cert = make_cert(ip)
        ssl = {"ssl_keyfile": key, "ssl_certfile": cert}
    scheme = "https" if use_https else "http"
    print(f"\n  TSUKUMO  ->  {scheme}://localhost:{port}   (phones: {scheme}://{ip}:{port})\n")
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info", **ssl)


if __name__ == "__main__":
    main()
