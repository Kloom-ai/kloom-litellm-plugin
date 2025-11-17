import asyncio
import os

import litellm

from kloom_litellm import KloomPlugin

# Set up the logger
kloom_logger = KloomPlugin()
litellm.callbacks = [kloom_logger]


async def test():
    # Make a completion - it will auto-log to Kloom
    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print(response)


asyncio.run(test())
