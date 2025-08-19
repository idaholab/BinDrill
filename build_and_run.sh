docker build -t binarydriller . && docker run -p 5567:5567 -v $(pwd):/binarydriller --name binarydriller binarydriller
