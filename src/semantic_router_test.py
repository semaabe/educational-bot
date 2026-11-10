from fastapi.testclient import TestClient
from backend.app import app  # Assuming 'app' is imported from here
from backend.auth import mint_token

client = TestClient(app)


# Real tokens signed with the API's per-process key, rather than mocking jwt.decode.
def bearer(role):
    return {"Authorization": f"Bearer {mint_token('test-user', role)}"}


ROLES = ["student", "professor"]

# Queries to test per role
test_queries = {
    "student": "I am feeling overwhelmed and need to talk to someone",
    "professor": "show me the progress report for my students",
}

def test_role_based_queries():
    for role in ROLES:
        query = test_queries[role]

        response = client.post(
            "/query",
            headers=bearer(role),
            json={
                "query": query,
                "guild_id": 123,
                "channel_id": 456
            }
        )
        
        # Print helpful debug info *before* the assertion
        print(f"\n=== TESTING ROLE: {role.upper()} ===")
        print(f"Query: \"{query}\"")
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")

        assert response.status_code == 200, f"{role} test failed with status {response.status_code}: {response.text}"
        
        data = response.json()
        assert "answer" in data, f"{role} response missing 'answer'key. Got: {data}"
        
        print(f"Result: PASSED")


def test_invalid_and_missing_tokens():
    # Test missing token (should return 401)
    response = client.post(
        "/query",
        json={"query": "any query", "guild_id": 0, "channel_id": 0}
    )
    assert response.status_code == 401, f"Missing token test failed: {response.status_code}"
    print("\n=== MISSING TOKEN RESPONSE ===")
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())

    # Test invalid token (should return 401)
    response = client.post(
        "/query",
        headers={"Authorization": "Bearer invalidtoken"},
        json={"query": "any query", "guild_id": 0, "channel_id": 0}
    )
    assert response.status_code in [401], f"Invalid token test failed: {response.status_code}"
    print("\n=== INVALID TOKEN RESPONSE ===")
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())

if __name__ == "__main__":
    print("Starting role-based query tests...")
    test_role_based_queries()
    print("\nStarting auth token tests...")
    test_invalid_and_missing_tokens()
    print("\nAll tests finished.")