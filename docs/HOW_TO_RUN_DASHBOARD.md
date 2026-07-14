# How to run the ALC Dashboard

This guide explains how to open the ALC Ecosystem Dashboard on your
computer. You do not need any programming experience.

---

## What you need

- The **alc_dashboard_demo.zip** file (about 20 MB)
- About **5 minutes** and an internet connection

> **Note about API tokens:** The dashboard itself does not need a token —
> it reads pre-processed data files included in the zip. If you want to
> re-run the full pipeline (extract fresh data from the KTool server),
> you will need a **bearer token**. See the section at the end of this
> guide for details.

---

## Step 1 — Install Python

The dashboard is a Python program. You need Python installed to run it.

### Windows

1. Open a web browser and go to https://www.python.org/downloads/
2. Click the yellow **Download Python** button (it will download a `.exe`
   file)
3. Open the downloaded file
4. **Important:** check the box that says **"Add Python to PATH"** at the
   bottom of the installer window
5. Click **Install Now**
6. When it finishes, close the installer

### Mac

1. Open a web browser and go to https://www.python.org/downloads/
2. Click the yellow **Download Python** button (it will download a `.pkg`
   file)
3. Open the downloaded file and follow the installer steps
4. When it finishes, close the installer

---

## Step 2 — Unzip the dashboard file

### Windows

1. Right-click on `alc_dashboard_demo.zip`
2. Select **Extract All**
3. Choose a location (your Desktop is fine)
4. Click **Extract**
5. A folder called `alc_dashboard_demo` will appear

### Mac

1. Double-click on `alc_dashboard_demo.zip`
2. A folder called `alc_dashboard_demo` will appear next to the zip file

---

## Step 3 — Open a terminal (command line)

A terminal is a window where you type text commands.

### Windows

1. Open the `alc_dashboard_demo` folder
2. Click in the **address bar** at the top of the folder window (it shows
   something like `This PC > ... > alc_dashboard_demo`)
3. Type `cmd` and press **Enter**
4. A black terminal window will open

### Mac

1. Open the **Terminal** app (press Cmd+Space, type "Terminal", press
   Enter)
2. Type `cd ` (with a space after it), then drag the
   `alc_dashboard_demo` folder from Finder onto the Terminal window
3. Press **Enter**

---

## Step 4 — Install the required packages

In the terminal, type the following command and press **Enter**:

```
python -m pip install -r requirements.txt
```

This will download and install the packages the dashboard needs. It might
take a minute or two. Wait until you see a new blinking cursor (>) in
the terminal.

**If you get an error on Windows** saying `'python' is not recognized`,
try this instead:

```
py -m pip install -r requirements.txt
```

**If you get an error on Mac**, try `python3` instead of `python`:

```
python3 -m pip install -r requirements.txt
```

---

## Step 5 — Launch the dashboard

In the same terminal, type this command and press **Enter**:

```
streamlit run app.py
```

After a few seconds, your web browser will open and show the dashboard.
If it does not open automatically, look in the terminal for a line that
says:

```
Local URL: http://localhost:8501
```

Copy and paste that address into your web browser.

---

## Step 6 — Using the dashboard

The dashboard starts with **Platform 173** (the real dataset).

To switch to the synthetic dataset:

1. Look at the left sidebar
2. In the **Platform ID** box, change `173` to `173_synthetic`
3. Press **Enter**
4. The dashboard will reload with the synthetic data

To switch back, change it back to `173`.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `'streamlit' is not recognized` | The packages didn't install correctly. Run Step 4 again. |
| Browser shows "ModuleNotFoundError" | Run Step 4 again and make sure there are no red error messages. |
| Map shows no data | Switch to `173_synthetic` in the sidebar — the real dataset has fewer locations mapped. |
| Dashboard looks empty | Try switching platform ID between `173` and `173_synthetic`. |
| Port 8501 already in use | In the terminal, press **Ctrl+C** (hold Control, press C), then run: `streamlit run app.py --server.port 8502` |

---

## Closing the dashboard

Close the browser tab. In the terminal, press **Ctrl+C** (Windows) or
**Control+C** (Mac) to stop the program. You can close the terminal
window after that.

---

## Bearer token for pipeline re-runs

The dashboard displays pre-computed data. If you want to re-run the
full analysis pipeline (extract fresh data from the KTool server and
recompute all metrics), you need a **bearer token** for API
authentication. No token is hardcoded in the code — it is read from an
environment variable.

### Where to find your token

1. Log in to your KTool instance as an administrator.
2. Open your browser's developer tools (F12 or right-click → Inspect).
3. Go to the **Application** tab (Chrome/Edge) or **Storage** tab (Firefox).
4. Look for **Local Storage** in the left sidebar.
5. Find an entry named `auth_token` or `token` or `jwt`.
6. Copy its value — it is a long string of letters, numbers, and dots.

### How to use it

**On Windows (Command Prompt):**
```
set KTOOL_AUTH_TOKEN=your_token_here
python src/analysis/MAIN_comprehensive_pipeline.py
```

**On Mac / Linux:**
```
export KTOOL_AUTH_TOKEN=your_token_here
python3 src/analysis/MAIN_comprehensive_pipeline.py
```

Replace `your_token_here` with the token you copied. The script will
fetch the latest data and rebuild the analysis files. After that,
launch the dashboard normally with `streamlit run app.py`.

### Security notes

- The token grants access to your KTool data. **Never share it** or
  commit it to a public repository.
- The pipeline scripts check the `KTOOL_AUTH_TOKEN` environment variable
  only — they will refuse to run if it is not set.
- If you are only viewing the pre-computed dashboard, you do **not**
  need a token at all.
