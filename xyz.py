from pinecone import Pinecone
pc = Pinecone(api_key="pcsk_5Lzm47_GcDtGcVXtNDpbc8jkvuJSvJusb9JTistqSdKhMTxfS4ySCnh2XfpvwNuGdzxUdJ")
index_info = pc.describe_index("textile-products")
print(index_info)
