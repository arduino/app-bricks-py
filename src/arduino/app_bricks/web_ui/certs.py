# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import os
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from datetime import datetime, timedelta, UTC


def cert_exists(root_dir: str) -> bool:
    """Check if the SSL certificate and private key files exist.

    Args:
        root_dir (str): The root directory where the SSL files are stored.

    Returns:
        bool: True if both key and cert files exist, False otherwise.
    """
    return os.path.exists(os.path.join(root_dir, "key.pem")) and os.path.exists(os.path.join(root_dir, "cert.pem"))


def get_cert(root_dir: str) -> str:
    """Get the path to the SSL certificate file.

    Args:
        root_dir (str): The root directory where the SSL files are stored.

    Returns:
        str: The path to the SSL certificate file.
    """
    return os.path.join(root_dir, "cert.pem")


def get_pkey(root_dir: str) -> str:
    """Get the path to the SSL private key file.

    Args:
        root_dir: The root directory where the SSL files are stored.

    Returns:
        str: The path to the SSL private key file.
    """
    return os.path.join(root_dir, "key.pem")


def generate_self_signed_cert(root_dir: str):
    """Generate a self-signed SSL certificate and private key.

    Args:
        root_dir (str): The root directory where the SSL files will be stored.
    """
    # Generate a private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Generate a self-signed certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IT"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Piedmont"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Turin"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Arduino"),
        x509.NameAttribute(NameOID.COMMON_NAME, "0.0.0.0"),
    ])
    cert = x509.CertificateBuilder()
    cert = cert.subject_name(subject)
    cert = cert.issuer_name(issuer)
    cert = cert.public_key(private_key.public_key())
    cert = cert.serial_number(x509.random_serial_number())
    cert = cert.not_valid_before(datetime.now(UTC))
    cert = cert.not_valid_after(datetime.now(UTC) + timedelta(days=365))  # Valid for 1 year
    cert = cert.add_extension(x509.SubjectAlternativeName([x509.DNSName("0.0.0.0")]), critical=False)
    cert = cert.sign(private_key, hashes.SHA256())

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Write the private key to a PEM file
    with open(os.path.join(root_dir, "key.pem"), "wb") as key_file:
        key_file.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write the certificate to a PEM file
    with open(os.path.join(root_dir, "cert.pem"), "wb") as cert_file:
        cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
