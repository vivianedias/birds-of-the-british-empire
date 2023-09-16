import dotenv from "dotenv-safe";
import {
  Document,
  ContextChatEngine,
  VectorStoreIndex,
  storageContextFromDefaults,
  SimpleDirectoryReader,
} from "llamaindex";

dotenv.config();

async function main() {
  const documents = await new SimpleDirectoryReader().loadData({
    directoryPath: "./data",
  });

  // Split text and create embeddings. Store them in a VectorStoreIndex
  // persist the vector store automatically with the storage context
  const storageContext = await storageContextFromDefaults({
    persistDir: "./storage",
  });
  const index = await VectorStoreIndex.fromDocuments(documents, {
    storageContext,
  });
  const retriever = index.asRetriever();
  const chatEngine = new ContextChatEngine({ retriever });

  // Query the index
  const response = await chatEngine.chat("Describe the readbreast trush", [
    {
      content:
        "Act like Charlotte Bronte, the 19th century English writer, and answer questions SOLELY based on the context provided to you. DO NOT use your previous knowledge. Copy Charlotte Bronte's writing style.",
      role: "system",
    },
  ]);

  // Output response
  console.log(response.toString());
}

main();
