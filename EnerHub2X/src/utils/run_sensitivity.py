import subprocess

co2_prices = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0]

for price in co2_prices:
    print(f"Running with CO2 price = {price}")

    cmd = [
        "python",
        "model_run.py",
        "--test",
        "--strategic",
        "--co2_market_price", str(price)
    ]

    subprocess.run(cmd, check=True)
