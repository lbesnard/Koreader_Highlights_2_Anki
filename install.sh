latest_release_url=$(curl -s https://api.github.com/repos/lbesnard/Koreader_Highlights_2_Anki/releases/latest | grep -o '"browser_download_url": ".*"' | grep -o 'https://.*.whl"')

# Extract the URL of the wheel file
latest_wheel_url=$(echo "$latest_release_url" | sed 's/"$//')

# Download the wheel file
curl -LO "$latest_wheel_url"

# Extract the filename from the URL
filename=$(basename "$latest_wheel_url")

# Install the package using pip
pip install "$filename"
