# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import *

client = HttpClient()

# --- Example GET request ---
successful_url_get = "https://jsonplaceholder.typicode.com/todos/1"
print(f"\n--- Testing a successful GET request: {successful_url_get} ---")
successful_response_get = client.request_with_retry(successful_url_get, method="GET")
if successful_response_get:
    print(f"Content (first 100 chars): {successful_response_get.text[:100]}...")
else:
    print("GET request failed.")

# --- Example POST request ---
post_url = "https://jsonplaceholder.typicode.com/posts"
post_data = {"title": "foo", "body": "bar", "userId": 1}
print(f"\n--- Testing a successful POST request to: {post_url} ---")
post_response = client.request_with_retry(post_url, method="POST", json=post_data)
if post_response:
    print(f"Content: {post_response.json()}")
else:
    print("POST request failed.")
