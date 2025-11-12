import numpy as np
import mmap
import os

class ChunkedMmapBuffer:
    def __init__(self, name, total_steps, step_size, chunk_size=500, prefix="exp_chunk"):
        self.total_steps = total_steps
        self.step_size = step_size
        self.chunk_size = chunk_size
        self.num_chunks = (total_steps + chunk_size - 1) // chunk_size
        self.name = name
        self.prefix = prefix

        # Хранить файлы, mmap-ы и numpy-массивы
        self.files = []
        self.mmaps = []
        self.arrays = []

        for i in range(self.num_chunks):
            self.this_chunk_size = min(chunk_size, total_steps - i * chunk_size)
            self.fname = f"{name}_{prefix}_{i}.dat"
            self.bytes_needed = self.this_chunk_size * step_size * 4
            # Создать и обнулить файл-чанк
            with open(self.fname, 'wb') as f:
                f.truncate(self.bytes_needed)

        self.is_chunks_created = False
        self.current_chunk = 0
           
    def open(self):
         # Открыть для записи-отображения
        for i in range(self.num_chunks):
            self.fname = f"{self.name}_{self.prefix}_{i}.dat"
            f = open(self.fname, "r+b")
            

            if not self.is_chunks_created:
                mm = mmap.mmap(f.fileno(), self.bytes_needed)
                arr = np.ndarray((self.this_chunk_size, self.step_size), dtype=np.float32, buffer=mm)
                self.files.append(f)
                self.mmaps.append(mm)
                self.arrays.append(arr)
        self.is_chunks_created = True

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        if chunk_idx > self.current_chunk:
            self.current_chunk += 1

        if chunk_idx >= self.num_chunks:
            raise IndexError("Index out of buffer bounds")
        return self.arrays[chunk_idx][local_idx]

    def __setitem__(self, idx, value):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        if chunk_idx >= self.num_chunks:
            raise IndexError("Index out of buffer bounds")
        self.arrays[chunk_idx][local_idx] = value

    def close(self, delete_files=False):
        for mm, f in zip(self.mmaps, self.files):
            mm.close()
            f.close()
        if delete_files:
            for i in range(self.num_chunks):
                fname = f"{self.name}_{self.prefix}_{i}.dat"
                os.remove(fname)

    @property
    def shape(self):
        return (self.total_steps, self.step_size)

    def __len__(self):
        return self.total_steps