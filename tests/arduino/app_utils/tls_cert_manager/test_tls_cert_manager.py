# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import shutil
import tempfile
import threading
import time
import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend

from arduino.app_utils.tls_cert_manager import TLSCertificateManager


@pytest.fixture
def temp_certs_dir():
    """Create a temporary directory for certificates and clean up after tests."""
    temp_dir = tempfile.mkdtemp()

    yield temp_dir

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def reset_manager():
    """Reset the TLSCertificateManager state between tests."""
    yield

    TLSCertificateManager._locks.clear()  # Reset state


class TestBasicFunctionality:
    """Test basic certificate creation and retrieval."""

    def test_create_certificates_in_custom_dir(self, temp_certs_dir, reset_manager):
        """Test creating certificates in a custom directory."""
        cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir)

        # Verify paths are correct
        assert cert_path == os.path.join(temp_certs_dir, "cert.pem")
        assert key_path == os.path.join(temp_certs_dir, "key.pem")

        # Verify files exist
        assert os.path.exists(cert_path)
        assert os.path.exists(key_path)

    def test_certificates_are_valid(self, temp_certs_dir, reset_manager):
        """Test that generated certificates are valid X.509 certificates."""
        cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir, common_name="test.local")

        # Load and verify certificate
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())

        # Check common name
        common_name = cert.subject.get_attributes_for_oid(x509.NameOID.COMMON_NAME)[0].value
        assert common_name == "test.local"

        # Check organization
        org = cert.subject.get_attributes_for_oid(x509.NameOID.ORGANIZATION_NAME)[0].value
        assert org == "Arduino"

    def test_reuse_existing_certificates(self, temp_certs_dir, reset_manager):
        """Test that existing certificates are reused instead of regenerated."""
        cert_path1, key_path1 = TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir)

        # Get modification time
        mtime1 = os.path.getmtime(cert_path1)

        # Get certificates again
        cert_path2, key_path2 = TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir)

        assert cert_path1 == cert_path2
        assert key_path1 == key_path2

        # Check modification time is unchanged
        mtime2 = os.path.getmtime(cert_path2)
        assert mtime1 == mtime2

    def test_custom_validity_period(self, temp_certs_dir, reset_manager):
        """Test creating certificates with custom validity period."""
        cert_path, _ = TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir, validity_days=1)

        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())

        validity_days = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
        assert validity_days == 1


class TestHelperMethods:
    """Test helper methods for checking and retrieving certificate paths."""

    def test_certificates_exist_returns_false_for_missing(self, temp_certs_dir, reset_manager):
        """Test certificates_exist returns False when certificates don't exist."""
        assert not TLSCertificateManager.certificates_exist(certs_dir=temp_certs_dir)

    def test_certificates_exist_returns_true_after_creation(self, temp_certs_dir, reset_manager):
        """Test certificates_exist returns True after certificates are created."""
        TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir)
        assert TLSCertificateManager.certificates_exist(certs_dir=temp_certs_dir)

    def test_get_certificate_path(self, temp_certs_dir, reset_manager):
        """Test get_certificate_path returns correct path."""
        expected_path = os.path.join(temp_certs_dir, "cert.pem")
        actual_path = TLSCertificateManager.get_certificate_path(certs_dir=temp_certs_dir)
        assert actual_path == expected_path

    def test_get_private_key_path(self, temp_certs_dir, reset_manager):
        """Test get_private_key_path returns correct path."""
        expected_path = os.path.join(temp_certs_dir, "key.pem")
        actual_path = TLSCertificateManager.get_private_key_path(certs_dir=temp_certs_dir)
        assert actual_path == expected_path


class TestConcurrentAccess:
    """Test concurrent access and race condition handling."""

    def test_concurrent_access_same_directory(self, temp_certs_dir, reset_manager):
        """Test multiple threads accessing the same directory concurrently."""
        results = []
        errors = []

        def create_certs(thread_id):
            try:
                start_time = time.time()
                cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir)
                elapsed = time.time() - start_time
                results.append({"thread_id": thread_id, "cert_path": cert_path, "key_path": key_path, "elapsed": elapsed})
            except Exception as e:
                errors.append({"thread_id": thread_id, "error": str(e)})

        # Start 10 threads simultaneously
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_certs, args=(i,))
            threads.append(thread)

        # Start all threads at once
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all threads got the same paths
        assert len(results) == 10
        cert_paths = set(r["cert_path"] for r in results)
        key_paths = set(r["key_path"] for r in results)
        assert len(cert_paths) == 1, "All threads should get the same certificate path"
        assert len(key_paths) == 1, "All threads should get the same key path"

        # Verify certificates exist and are valid
        cert_path = results[0]["cert_path"]
        assert os.path.exists(cert_path)
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read(), default_backend())
            assert cert is not None

    def test_concurrent_access_different_directories(self, temp_certs_dir, reset_manager):
        """Test multiple threads accessing different directories concurrently."""
        results = []
        errors = []

        def create_certs(component_name):
            try:
                start_time = time.time()
                component_dir = os.path.join(temp_certs_dir, component_name)
                cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=component_dir)
                elapsed = time.time() - start_time
                results.append({"component": component_name, "cert_path": cert_path, "key_path": key_path, "elapsed": elapsed})
            except Exception as e:
                errors.append({"component": component_name, "error": str(e)})

        # Simulate multiple components starting simultaneously
        components = ["webui", "api", "mqtt", "scanner", "processor"]
        threads = []

        for component in components:
            thread = threading.Thread(target=create_certs, args=(component,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all components succeeded
        assert len(results) == len(components)

        # Verify each component has its own certificates
        cert_dirs = set(os.path.dirname(r["cert_path"]) for r in results)
        assert len(cert_dirs) == len(components), "Each component should have its own directory"

        # Verify all certificates exist and are in correct directories
        for result in results:
            component = result["component"]
            expected_dir = os.path.join(temp_certs_dir, component)
            assert expected_dir in result["cert_path"]
            assert os.path.exists(result["cert_path"])
            assert os.path.exists(result["key_path"])

    def test_concurrent_mixed_access(self, temp_certs_dir, reset_manager):
        """Test concurrent access with both shared and component-specific directories."""
        results = []
        errors = []
        lock = threading.Lock()

        def create_certs(name, use_custom_dir):
            try:
                start_time = time.time()
                if use_custom_dir:
                    certs_dir = os.path.join(temp_certs_dir, name)
                else:
                    certs_dir = temp_certs_dir

                cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=certs_dir)
                elapsed = time.time() - start_time

                with lock:
                    results.append({"name": name, "use_custom": use_custom_dir, "cert_path": cert_path, "elapsed": elapsed})
            except Exception as e:
                with lock:
                    errors.append({"name": name, "error": str(e)})

        # Mix of shared and custom directory access
        configs = [
            ("webui", False),  # Shared
            ("api", False),  # Shared
            ("mqtt", True),  # Custom
            ("scanner", True),  # Custom
            ("backup", False),  # Shared
            ("processor", True),  # Custom
        ]

        threads = []
        for name, use_custom in configs:
            thread = threading.Thread(target=create_certs, args=(name, use_custom))
            threads.append(thread)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == len(configs)

        # Verify shared components use the same certificates
        shared_certs = [r for r in results if not r["use_custom"]]
        shared_paths = set(r["cert_path"] for r in shared_certs)
        assert len(shared_paths) == 1, "Shared components should use same certificates"

        # Verify custom components have unique certificates
        custom_certs = [r for r in results if r["use_custom"]]
        custom_paths = set(r["cert_path"] for r in custom_certs)
        assert len(custom_paths) == len(custom_certs), "Custom components should have unique certificates"


class TestDirectoryCreation:
    """Test automatic directory creation."""

    def test_creates_missing_directory(self, temp_certs_dir, reset_manager):
        """Test that missing directories are created automatically."""
        nested_dir = os.path.join(temp_certs_dir, "deeply", "nested", "path")
        assert not os.path.exists(nested_dir)

        cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=nested_dir)

        assert os.path.exists(nested_dir)
        assert os.path.exists(cert_path)
        assert os.path.exists(key_path)

    def test_handles_existing_directory(self, temp_certs_dir, reset_manager):
        """Test that existing directories are handled correctly."""
        # Pre-create the directory
        os.makedirs(temp_certs_dir, exist_ok=True)

        cert_path, key_path = TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir)

        assert os.path.exists(cert_path)
        assert os.path.exists(key_path)


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_directory_permissions(self, reset_manager):
        """Test handling of directories with invalid permissions."""
        # This test is platform-specific and may need adjustment
        if os.name != "posix":
            pytest.skip("Permission test only applicable on POSIX systems")

        temp_dir = tempfile.mkdtemp()
        try:
            # Make directory read-only
            os.chmod(temp_dir, 0o444)

            with pytest.raises(RuntimeError) as exc_info:
                TLSCertificateManager.get_or_create_certificates(certs_dir=temp_dir)

            assert "Failed to generate TLS certificates" in str(exc_info.value)
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_dir, 0o755)
            shutil.rmtree(temp_dir)


class TestPerformance:
    """Test performance characteristics."""

    def test_fast_path_no_lock_overhead(self, temp_certs_dir, reset_manager):
        """Test that retrieving existing certificates is fast (no lock acquisition)."""
        # Create certificates first
        TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir)

        # Measure retrieval time
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            TLSCertificateManager.get_or_create_certificates(certs_dir=temp_certs_dir)
        elapsed = time.time() - start

        # Should be very fast (< 1ms per call on average)
        avg_time = elapsed / iterations
        assert avg_time < 0.001, f"Average retrieval time too slow: {avg_time:.6f}s"

    def test_concurrent_different_dirs_no_blocking(self, temp_certs_dir, reset_manager):
        """Test that different directories don't block each other significantly."""
        total_elapsed_lock = threading.Lock()
        elapsed_times = []

        def create_certs(brick_name):
            brick_dir = os.path.join(temp_certs_dir, brick_name)

            start = time.time()
            TLSCertificateManager.get_or_create_certificates(certs_dir=brick_dir)
            elapsed = time.time() - start

            with total_elapsed_lock:
                elapsed_times.append(elapsed)

        bricks = ["brick1", "brick2", "brick3", "brick4"]
        threads = [threading.Thread(target=create_certs, args=(c,)) for c in bricks]

        start = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        overall_run_time = time.time() - start

        # If bricks truly run in parallel, total time should be lower than
        # the cumulative total run times by all threads
        assert overall_run_time < sum(elapsed_times), (
            f"Bricks blocked each other: {overall_run_time:.3f}s should be lower than {sum(elapsed_times):.3f}s"
        )
