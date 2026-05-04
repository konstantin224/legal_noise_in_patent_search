docker run -d --name local_qdrant ^
  -p 6333:6333 -p 6334:6334 ^
  -v C:\Users\rudkevich-k\Desktop\local_qdrant\storage:/qdrant/storage ^
  -v C:\Users\rudkevich-k\Desktop\local_qdrant\snapshots:/qdrant/snapshots ^
  qdrant/qdrant:latest