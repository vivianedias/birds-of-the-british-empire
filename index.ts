import dotenv from "dotenv-safe";
import {
  ContextChatEngine,
  VectorStoreIndex,
  storageContextFromDefaults,
  SimpleDirectoryReader,
} from "llamaindex";
import OpenAI from "openai";

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
  const response = await chatEngine.chat(
    "Describe the white wagtail appearance. Don't mention its name.",
    [
      {
        content:
          "You are Charlotte Bronte, the 19th-century English writer. Answer questions SOLELY based on the context provided to you. DO NOT use your previous knowledge. Copy Charlotte Bronte's writing style from her book 'Jane Eyre' and respond like you are providing a prompt to Open AI's DALL-E image generator. Your answers should have a maximum of 1000 characters total.",
        role: "system",
      },
    ]
  );

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const imageResponse = await openai.images.generate({
    prompt: response.toString(),
    n: 3,
    size: "1024x1024",
  });

  // Output response
  console.log(response.toString());
  console.log(imageResponse);
}

main();
