#!/usr/bin/env python3
"""
ACE-Step LoRA Trainer + Captioner — Launcher

Usage:
    python launch.py                    # Launch training UI (default)
    python launch.py --mode train       # Launch training UI
    python launch.py --mode caption     # Launch captioner UI
    python launch.py --mode both        # Launch both on different ports
"""

import argparse
import sys
import os


def check_environment():
    """Check that we're running in a proper environment with all dependencies."""
    # Check if we're in a virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

    if not in_venv:
        print("\n" + "=" * 60)
        print("  ⚠️  WARNING: Not running inside a virtual environment!")
        print("=" * 60)
        print()
        if sys.platform == 'win32':
            print("  Please run install.bat first, then use start.bat")
            print("  Or manually:")
            print("    env\\Scripts\\activate && python launch.py")
        else:
            print("  Please run install.sh first, then use start.sh")
            print("  Or manually:")
            print("    source env/bin/activate && python launch.py")
        print()
        print("  Without a virtual environment, critical packages like")
        print("  PEFT may be missing, causing training to run WITHOUT")
        print("  LoRA (full fine-tuning instead — much slower and larger).")
        print("=" * 60)
        print()
        response = input("  Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            sys.exit(0)
        print()

    # Check critical packages
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch (PyTorch)")

    try:
        import peft
    except ImportError:
        missing.append("peft (CRITICAL — without this, LoRA training is disabled!)")

    try:
        import lightning
    except ImportError:
        missing.append("lightning")

    try:
        import gradio
    except ImportError:
        missing.append("gradio")

    if missing:
        print("\n" + "=" * 60)
        print("  ❌  MISSING PACKAGES DETECTED")
        print("=" * 60)
        for pkg in missing:
            print(f"  • {pkg}")
        print()
        if sys.platform == 'win32':
            print("  Run install.bat to install all dependencies.")
        else:
            print("  Run install.sh to install all dependencies.")
        print("=" * 60)
        print()
        response = input("  Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            sys.exit(0)
        print()


def launch_training_ui(host="127.0.0.1", port=7861, share=False):
    """Launch the LoRA Training UI."""
    from lora_training_ui import create_ui
    from loguru import logger

    logger.info(f"Starting ACE-Step LoRA Training UI on {host}:{port}")
    demo = create_ui()
    demo.queue()

    allowed_paths = ["/", "C:\\", "D:\\", "E:\\", "F:\\", "G:\\", "H:\\"]
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        allowed_paths=allowed_paths,
    )


def launch_captioner_ui(host="127.0.0.1", port=7862, share=False):
    """Launch the Captioner UI."""
    from captioner_standalone import create_captioner_ui
    from loguru import logger

    logger.info(f"Starting ACE-Step Captioner UI on {host}:{port}")
    demo = create_captioner_ui()
    demo.queue()

    allowed_paths = ["/", "C:\\", "D:\\", "E:\\", "F:\\", "G:\\", "H:\\"]
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        allowed_paths=allowed_paths,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ACE-Step LoRA Trainer + Captioner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                        # Training UI on port 7861
  python launch.py --mode caption         # Captioner UI on port 7862
  python launch.py --mode both            # Both UIs
  python launch.py --port 8080            # Custom port
  python launch.py --share                # Create public Gradio link
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "caption", "both"],
        default="train",
        help="Which UI to launch (default: train)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7861, help="Port (default: 7861)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    args = parser.parse_args()

    if args.mode == "train":
        launch_training_ui(args.host, args.port, args.share)
    elif args.mode == "caption":
        launch_captioner_ui(args.host, args.port, args.share)
    elif args.mode == "both":
        import threading

        # Training on main thread, captioner on background thread
        captioner_port = args.port + 1
        print(f"\n{'='*50}")
        print(f"  Training UI:  http://{args.host}:{args.port}")
        print(f"  Captioner UI: http://{args.host}:{captioner_port}")
        print(f"{'='*50}\n")

        t = threading.Thread(
            target=launch_captioner_ui,
            args=(args.host, captioner_port, args.share),
            daemon=True,
        )
        t.start()
        launch_training_ui(args.host, args.port, args.share)


if __name__ == "__main__":
    check_environment()
    main()
