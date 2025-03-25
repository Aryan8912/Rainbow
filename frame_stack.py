import numpy as np

class FrameStack:
    def __init__(self, dimension=1, mode="array"):
        self.frames = None
        self.mode = mode
        self.dimension = dimension
        self.curr_frames = 0
        self.w = None
        self.h = None
        self.c = None

    def unstack(self):
        assert self.w is not None and self.curr_frames > 0
        if self.mode == "array":
            self.frames = self.frames[self.w:]
        if self.mode == "pixels":
            self.frames = self.frames[self.h:]
        self.curr_frames -= 1

    def stack(self, img, repeat=1):

        assert 0 <= repeat
        for _ in range(repeat):
            if self.mode == "array":
                if self.frames == 0:
                    self.w = img.shape[0]
                    self.frames = img
                else:
                    self.frames = np.concatenate((self.frames, img), axis=-1)
            if self.mode == "pixels":
                if self.curr_frames == 0:
                    self.w = img.shape[1]
                    self.h = img.shape[0]
                    self.c = img.shape[2]
                    self.frames = img
            else: 
                self.frames = np.concatenate((self.frames, img), axis=0)
        self.curr_frames += 1
        if self.curr_frames > self.dimension:
            self.unstack()
    return self.frames

    def clear(self):
        while self.curr_frames > 0:
            self.unstack()

    def full(self):
        return self.curr_frames == self.dimension
    
    def __len__(self):
        return self.curr_frames
    
    def get_frames(self):
        return self.frames
    
if __name__ == '__main__':
    m = FrameStack(4, mode='pixels')
    f = []
    for i in range(10):
        f.append(np.random.uniform(size=[12, 10, 3]))
    m.stack(f[0])
    print(len(m))
    m.stack(f[1], 8) 
    print(len(m))
    m.clear()
    print(len(m))       