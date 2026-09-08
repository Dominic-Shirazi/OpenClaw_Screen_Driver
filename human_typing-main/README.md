# Human Typer

A Python project that realistically simulates human typing behavior. Rather than instantly pasting or typing text at robotic speeds, Human Typer mimics how a real person types on a keyboard. It intelligently models timing variations, fatigue, common typing mistakes (and corrections), and muscle memory acceleration.

This tool is designed to be called by external scripts to enter text into any focused window as if a human was sitting at the keyboard typing it out.

## Features

- **Burst Typing Cadence**: Designed for fast typists, it simulates "bursting" through known words at high speeds (muscle memory) and pausing longer between words (cognitive planning), accurately maintaining the overall target WPM.
- **Accuracy Modeling (Mistakes & Corrections)**: Intentionally makes mistakes based on a configurable accuracy percentage. When it errs, it simulates hitting an adjacent "QWERTY neighbor" key, pausing to realize the mistake, pressing backspace, and correcting it.
- **Bigram Acceleration**: Common sequences and suffixes (e.g., "tion", "ing", "th", "and") trigger "muscle memory," typing them up to 70% faster than the base delay.
- **Fatigue Modeling**: The delay between keystrokes subtly increases the longer a continuous block of text is typed, compounding over time to emulate finger fatigue.
- **Mechanical Penalties**: Introduces slightly longer delays when holding Shift for uppercase letters, or when typing the same letter twice consecutively (which requires more precise finger timing).
- **Cognitive Pauses**: Adds varied micro-pauses for punctuation (commas, periods, semicolons, etc.) to mimic the natural cadence of thinking while typing.

## How It Works

The core of the logic is inside the `HumanTyper` class found in `human_typer.py`:
1. It calculates a baseline delay between keystrokes from the provided Target WPM.
2. It iterates through the string character by character, first checking for fast-typing sequences (like "ing" or "tion").
3. For standard characters, it rolls against your configured Accuracy. If it fails the check, it refers to its internal QWERTY adjacency map to deliberately strike the wrong nearby key.
4. If a mistake is made, it simulates a short reaction-time pause, sends a backspace command, and then types the correct character.
5. All typing commands are handled by the [`keyboard`](https://github.com/boppreh/keyboard) library, meaning strokes happen at the OS level and work in whatever application is currently focused.

## Configuration

Settings are controlled via `config.yaml` placed in the same directory:

```yaml
typing:
  wpm: 105       # Target typing speed in Words Per Minute
  accuracy: 0.96 # Percentage of correct keystrokes (0.0 to 1.0)
```

Lower the accuracy closer to `0.85` or `0.90` for a more error-prone, realistic look. Increase WPM to type faster.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd human_typing
   ```

2. **Install the required dependencies:**
   This project relies on `keyboard` and `PyYAML`. Install them via pip:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Because the `keyboard` library hooks into the operating system at a low level to simulate physical keystrokes, you may need to run your script with Administrator privileges on Windows (or root on Linux, accessibility permissions on macOS).*

## Usage

### As a Standalone Test
You can run the script directly to see a quick demonstration of the typing in action.
```bash
python human_typer.py
```
*You will have 3 seconds to focus your cursor into a text field before the script begins typing.*

### As an Imported Module
Integrating Human Typer into your own scripts is straightforward:

```python
from human_typer import HumanTyper

# Initialize the typer (it automatically looks for config.yaml in the same directory)
typer = HumanTyper()

# Start typing
typer.type("This text will be typed naturally with human-like delays and potential errors.")
```
