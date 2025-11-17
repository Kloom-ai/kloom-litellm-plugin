# Kloom LiteLLM Example

To run, you will need the following environment variables:

```
KLOOM_PROJECT_ID
KLOOM_API_KEY (set to your personal access token)
KLOOM_BASE_URL (optional; default is https://api.kloom.ai)
OPENAI_API_KEY
```

To get your project ID, run the following:

```
curl -H "Authorization: Bearer $KLOOM_API_KEY" https://api.kloom.ai/user-dashboard/list-projects
```

Then, you can run the example via (from the directory above this one):

```
uv run python example/litellm_local.py
```

And you will see a new entry in your History page.
