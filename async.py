import asyncio


async def generate_data(name, n, sleep):
    for x in range(n):
        await asyncio.sleep(sleep)
        print(f'{name}:{x}')
    return name


async def main():
    a, b = generate_data('a', 3, 1), generate_data('b', 2, 1)
    z = await asyncio.gather(a, b)
    print(z)




asyncio.get_event_loop().run_until_complete(main())
