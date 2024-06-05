# Setup

Create a virtual environment and install the required Python and Julia packages:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import juliapkg; juliapkg.resolve()"
```

# Example
You can check out the code in `example.py` for how to use the operator `(I - XG)` (that I call "LippmannSchwinger" in the code):
```bash
python -i example.py
```
