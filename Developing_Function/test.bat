chcp 65001
cd C:\Users\ASUS\OneDrive - 長庚大學\桌面\專題程式\音樂生成\開發\Developing_Function\fluid\bin
start fluidsynth -T flac -F test0.flac test.sf2 intro.mid
start fluidsynth -ni test.sf2 intro.mid -F output.wav -r 44100
cd C:\Users\ASUS\OneDrive - 長庚大學\桌面\專題程式\音樂生成\開發\Developing_Function