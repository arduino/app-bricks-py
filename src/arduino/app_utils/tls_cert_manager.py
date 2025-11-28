# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import threading
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from datetime import datetime, timedelta, UTC


DEFAULT_CERTS_DIR = "/app/certs"
DEFAULT_CERTS_PARAMS = {
    "country_name": "IT",
    "state_or_province_name": "Piedmont",
    "locality_name": "Turin",
    "organization_name": "Arduino",
    "common_name": "0.0.0.0",
    "validity_days": 365,
}


class TLSCertificateManager:
    """Certificate manager for TLS certificates.

    This class handles certificate generation and retrieval on a Brick basis. By default, all bricks
    share certificates from the default directory (/app/certs).
    Components can use their own certificates by providing a different certs_dir path.
    """

    _locks = {}
    _locks_lock = threading.Lock()

    @classmethod
    def get_or_create_certificates(
        cls,
        certs_dir: str = DEFAULT_CERTS_DIR,
        country_name: str = DEFAULT_CERTS_PARAMS["country_name"],
        state_or_province_name: str = DEFAULT_CERTS_PARAMS["state_or_province_name"],
        locality_name: str = DEFAULT_CERTS_PARAMS["locality_name"],
        organization_name: str = DEFAULT_CERTS_PARAMS["organization_name"],
        common_name: str = DEFAULT_CERTS_PARAMS["common_name"],
        validity_days: int = DEFAULT_CERTS_PARAMS["validity_days"],
    ) -> tuple[str, str]:
        """Get or create TLS certificates at the specified path.

        By default, uses shared certificates in /app/certs. If a different certs_dir is provided,
        uses certificates specific to that directory (useful for brick-specific certificates).

        Concurrent access is managed to prevent race conditions when multiple bricks attempt to
        access certificates simultaneously.

        Args:
            certs_dir (str, optional): Directory for certificates. Defaults to /app/certs (shared
                by all bricks). Provide a different path for brick-specific certificates.
            country_name (str, optional): Country name for the certificate. Defaults to "IT".
            state_or_province_name (str, optional): State or province name for the certificate.
                Defaults to "Piedmont".
            locality_name (str, optional): Locality name for the certificate. Defaults to "Turin".
            organization_name (str, optional): Organization name for the certificate. Defaults to "Arduino".
            common_name (str, optional): Common name for the certificate. Defaults to "0.0.0.0".
            validity_days (int, optional): Certificate validity period in days. Defaults to 365.

        Returns:
            tuple[str, str]: Paths to (certificate_file, private_key_file)

        Raises:
            RuntimeError: If certificate generation fails.
        """
        target_dir = certs_dir or DEFAULT_CERTS_DIR
        cert_path = os.path.join(target_dir, "cert.pem")
        key_path = os.path.join(target_dir, "key.pem")

        if cls.certificates_exist(target_dir):
            return cert_path, key_path

        dir_lock = cls._get_dir_lock(target_dir)
        with dir_lock:
            if cls.certificates_exist(target_dir):
                return cert_path, key_path

            try:
                cls._generate_self_signed_cert(
                    target_dir, country_name, state_or_province_name, locality_name, organization_name, common_name, validity_days
                )
                return cert_path, key_path
            except Exception as e:
                raise RuntimeError(f"Failed to generate TLS certificates in {target_dir}: {e}") from e

    @classmethod
    def certificates_exist(cls, certs_dir: str = DEFAULT_CERTS_DIR) -> bool:
        """Check if TLS certificates exist in the given directory.

        Args:
            certs_dir (str, optional): Directory for certificates.
                Defaults to /app/certs.

        Returns:
            bool: True if both certificate and key files exist, False otherwise.
        """
        target_dir = certs_dir or DEFAULT_CERTS_DIR
        cert_path = os.path.join(target_dir, "cert.pem")
        key_path = os.path.join(target_dir, "key.pem")
        return os.path.exists(cert_path) and os.path.exists(key_path)

    @classmethod
    def get_certificates_paths(cls, certs_dir: str = DEFAULT_CERTS_DIR) -> tuple[str, str]:
        """Get the paths to the TLS certificate and private key files.

        Args:
            certs_dir (str, optional): Directory for certificates. Defaults to /app/certs.
        Returns:
            tuple[str, str]: Paths to certificate_file and private_key_file
        """
        target_dir = certs_dir or DEFAULT_CERTS_DIR
        return cls.get_certificate_path(target_dir), cls.get_private_key_path(target_dir)

    @classmethod
    def get_certificate_path(cls, certs_dir: str = DEFAULT_CERTS_DIR) -> str:
        """Get the path to the TLS certificate file.

        Args:
            certs_dir (str, optional): Directory for certificates. Defaults to /app/certs.

        Returns:
            str: Path to the certificate file.
        """
        return os.path.join(certs_dir or DEFAULT_CERTS_DIR, "cert.pem")

    @classmethod
    def get_private_key_path(cls, certs_dir: str = DEFAULT_CERTS_DIR) -> str:
        """Get the path to the TLS private key file.

        Args:
            certs_dir (str, optional): Directory for certificates. Defaults to /app/certs.

        Returns:
            str: Path to the private key file.
        """
        return os.path.join(certs_dir or DEFAULT_CERTS_DIR, "key.pem")

    @classmethod
    def _get_dir_lock(cls, target_dir: str) -> threading.Lock:
        """Get or create a lock for a specific directory.

        This ensures that only operations on the same directory block each other,
        while operations on different directories can proceed concurrently.

        Args:
            target_dir (str): The normalized absolute path to the directory.

        Returns:
            threading.Lock: A lock specific to this directory.
        """
        with cls._locks_lock:
            if target_dir not in cls._locks:
                cls._locks[target_dir] = threading.Lock()
            return cls._locks[target_dir]

    @staticmethod
    def _generate_self_signed_cert(
        target_dir: str,
        country_name: str,
        state_or_province_name: str,
        locality_name: str,
        organization_name: str,
        common_name: str,
        validity_days: int,
    ):
        # Generate a private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Generate a self-signed certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, country_name),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, state_or_province_name),
            x509.NameAttribute(NameOID.LOCALITY_NAME, locality_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization_name),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])

        cert = x509.CertificateBuilder()
        cert = cert.subject_name(subject)
        cert = cert.issuer_name(issuer)
        cert = cert.public_key(private_key.public_key())
        cert = cert.serial_number(x509.random_serial_number())
        cert = cert.not_valid_before(datetime.now(UTC))
        cert = cert.not_valid_after(datetime.now(UTC) + timedelta(days=validity_days))
        cert = cert.add_extension(x509.SubjectAlternativeName([x509.DNSName(common_name)]), critical=False)
        cert = cert.sign(private_key, hashes.SHA256())

        Path(target_dir).mkdir(parents=True, exist_ok=True)

        # Write the certificate to a PEM file
        cert_path = os.path.join(target_dir, "cert.pem")
        with open(cert_path, "wb") as cert_file:
            cert_file.write(cert.public_bytes(serialization.Encoding.PEM))

        # Write the private key to a PEM file
        key_path = os.path.join(target_dir, "key.pem")
        with open(key_path, "wb") as key_file:
            key_file.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
