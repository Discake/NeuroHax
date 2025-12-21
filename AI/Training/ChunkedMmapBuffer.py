import numpy as np
import mmap
import os

class ChunkedMmapBuffer:
    def create(self, name, total_steps, step_size, chunk_size=500, prefix="exp_chunk"):
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
           
    def open(self, total_steps, step_size, shm_name):
         # Открыть для записи-отображения
        chunk_size = 500
        self.chunk_size = chunk_size
        self.num_chunks = (total_steps + chunk_size - 1) // chunk_size
        self.is_chunks_created = False

         # Хранить файлы, mmap-ы и numpy-массивы
        self.files = []
        self.mmaps = []
        self.arrays = []

        prefix="exp_chunk"
        self.prefix = prefix

        for i in range(self.num_chunks):
            self.this_chunk_size = min(chunk_size, total_steps - i * chunk_size)
            self.bytes_needed = self.this_chunk_size * step_size * 4


            self.fname = shm_name
            f = open(self.fname + f"_{prefix}_{i}.dat", "r+b")
            

            if not self.is_chunks_created:
                mm = mmap.mmap(f.fileno(), self.bytes_needed)
                arr = np.ndarray((self.this_chunk_size, step_size), dtype=np.float32, buffer=mm)
                self.files.append(f)
                self.mmaps.append(mm)
                self.arrays.append(arr)
        self.is_chunks_created = True

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        if chunk_idx >= self.num_chunks:
            raise IndexError("Index out of buffer bounds")
        return self.arrays[chunk_idx][local_idx]

    def __setitem__(self, idx, value):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        if chunk_idx >= self.num_chunks:
            raise IndexError("Index out of buffer bounds")
        self.arrays[chunk_idx][local_idx] = value

    def close(self, shm_name, delete_files=False, clear_files = True):
        for mm, f in zip(self.mmaps, self.files):
            mm.close()
            f.close()
        if delete_files:
            for i in range(self.num_chunks):
                fname = f"{shm_name}_{self.prefix}_{i}.dat"
                os.remove(fname)
        if clear_files:
            for i in range(self.num_chunks):
                fname = f"{shm_name}_{self.prefix}_{i}.dat"
                with open(fname, 'r+b') as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    f.seek(0)
                    f.write(b'\x00' * size)
                    f.flush()

    @property
    def shape(self):
        return (self.total_steps, self.step_size)

    def __len__(self):
        return self.total_steps