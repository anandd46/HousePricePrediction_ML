"""
house_price_prediction.py
--------------------------
Top-level entry point for the House Price Prediction project.

This script provides a simple command-line menu so a user can run the
entire project without remembering individual file names.

Usage:
    python house_price_prediction.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def print_banner() -> None:
    """Display the project banner and menu."""
    print("=" * 60)
    print("🏠  HOUSE PRICE PREDICTION USING MACHINE LEARNING")
    print("=" * 60)
    print("1. Train models (preprocessing + training + evaluation + plots)")
    print("2. Predict a house price using the saved best model")
    print("3. Exit")
    print("=" * 60)


def run_script(script_name: str) -> None:
    """
    Run another Python script in the project as a subprocess, streaming
    its output directly to the console.

    Args:
        script_name: Filename of the script to run (e.g. 'train_model.py').
    """
    if not os.path.exists(script_name):
        print(f"❌ Could not find '{script_name}' in the current directory.")
        return

    subprocess.run([sys.executable, script_name], check=False)


def main() -> None:
    """Run the interactive project menu loop."""
    while True:
        print_banner()
        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            run_script("train_model.py")
        elif choice == "2":
            if not os.path.exists("saved_model.pkl"):
                print("⚠️  No trained model found. Please run option 1 first.\n")
                continue
            run_script("predict.py")
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("⚠️  Invalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()
