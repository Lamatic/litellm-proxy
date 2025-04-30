import pytest
import requests
import os
import json

# Get the service URL from environment variable or use default for testing
SERVICE_URL = os.environ.get('SERVICE_URL', 'https://litellm-testing-71891047326.us-central1.run.app')

class TestLiteLLM:
    def test_openai_chat_completions(self):
        response = requests.post(
            f"{SERVICE_URL}/chat/completions",
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': 'Bearer sk-1234'
            },
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Just say hi"
                    }
                ],
                "max_tokens": 1000,
                "api_key": os.environ.get('OPENAI_API_KEY'),
                "model": "gpt-4o-mini"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        
    def test_embeddings(self):
        response = requests.post(
            f"{SERVICE_URL}/embeddings",
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': 'Bearer sk-1234'
            },
            json={
                "input": ["hello"],
                "api_key": os.environ.get('OPENAI_API_KEY'),
                "model": "text-embedding-ada-002"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        
    def test_image_generation(self):
        response = requests.post(
            f"{SERVICE_URL}/images/generations",
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json',
                'Authorization': 'Bearer sk-1234'
            },
            json={
                "model": "dall-e-2",
                "prompt": "a white siamese cat",
                "n": 1,
                "size": "512x512",
                "api_key": os.environ.get('OPENAI_API_KEY')
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0