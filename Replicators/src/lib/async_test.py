import asyncio

async def serve_epoch(epoch):
    for i in range(epoch, epoch + 5):
        yield i
        await asyncio.sleep(0)  # This line is not necessary for the generator to work, but it allows other tasks to run

async def serve():
    epoch = 0
    svr = serve_epoch(epoch)
    try:
        return await svr.__anext__()
    except StopAsyncIteration:
        return None

# Usage
async def main():
    value = await serve()
    print(value)

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())