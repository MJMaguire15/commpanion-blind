param([int]$MaxIndex = 6)

for ($i=0; $i -le $MaxIndex; $i++) {
  $cmd = "import cv2,sys; i=int(sys.argv[1]); import cv2; cap=cv2.VideoCapture(i, cv2.CAP_DSHOW); ok=cap.isOpened(); print(f'index {i} -> open={ok}'); cap.release()"
  & python -c $cmd $i
}
