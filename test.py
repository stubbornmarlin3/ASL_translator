import numpy as np

t = np.random.randint(0,255,(500,500))
print(t)

with open("./test.gz", "wb") as f:
    x = t.flatten()
    for i in x:
        f.write(int(i).to_bytes())

with open("./test.gz", "rb") as f:
    frames = []
    while frame := [byte for byte in f.read(500)]:
        frames.append(frame)

    print(t == np.array(frames))
