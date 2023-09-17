import dotenv from "dotenv-safe";
import {
  ContextChatEngine,
  VectorStoreIndex,
  storageContextFromDefaults,
  SimpleDirectoryReader,
  OpenAI as LlamaOpenAI,
} from "llamaindex";
import OpenAI from "openai";

dotenv.config();

async function getVectorIndex(subDir: string) {
  const documents = await new SimpleDirectoryReader().loadData({
    directoryPath: `./data/${subDir}`,
  });

  // Split text and create embeddings. Store them in a VectorStoreIndex
  // persist the vector store automatically with the storage context
  const storageContext = await storageContextFromDefaults({
    persistDir: `./storage/${subDir}`,
  });
  console.log("generating vector index for:", subDir);
  const index = await VectorStoreIndex.fromDocuments(documents, {
    storageContext,
  });
  console.log("done");
  return index;
}

async function getChatEngine(index: VectorStoreIndex) {
  const retriever = index.asRetriever();
  const chatEngine = new ContextChatEngine({ retriever });

  return chatEngine;
}

async function main() {
  const fieldGuideIndex = await getVectorIndex("field-guide");
  const fieldGuideChatEngine = await getChatEngine(fieldGuideIndex);
  const FIELD_GUIDE_QUERY =
    "Describe the white wagtail appearance. Don't mention its name.";
  const queryFieldGuide = await fieldGuideChatEngine.chat(FIELD_GUIDE_QUERY, [
    {
      content:
        "Answer questions SOLELY based on the context provided to you. DO NOT use your previous knowledge.",
      role: "system",
    },
  ]);
  const fieldGuideDescription = queryFieldGuide.toString();
  console.log("field guide description:", fieldGuideDescription);

  const fictionAuthorIndex = await getVectorIndex("fiction-author");
  const fictionAuthorChatEngine = await getChatEngine(fictionAuthorIndex);
  const FICTION_AUTHOR_QUERY = `Rewrite the following sentence based on the author you should impersonate:
  '''
  ${fieldGuideDescription}
  '''
  `;
  const queryFictionAuthor = await fictionAuthorChatEngine.chat(
    FICTION_AUTHOR_QUERY,
    [
      {
        content:
          "You are Charlotte Bronte, the 19th-century English writer. Answer questions copying Charlotte Bronte's writing style from her book 'Jane Eyre' that's on the context provided to you.",
        role: "system",
      },
    ]
  );
  const fictionAuthorDescription = queryFictionAuthor.toString();
  console.log("author description:", fictionAuthorDescription);

  const llm = new LlamaOpenAI({ model: "gpt-3.5-turbo", temperature: 0.0 });
  const response = await llm.chat([
    {
      content: `You are an assistant that formats queries to Open AI's DALL-E image generator prompt. Your response needs to have a max of 1000 characters.`,
      role: "system",
    },
    { content: fictionAuthorDescription, role: "user" },
  ]);

  console.log("image generator prompt:", response.message.content);

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
  const imageResponse = await openai.images.generate({
    prompt: response.message.content,
    n: 3,
    size: "1024x1024",
  });

  console.log(imageResponse);
}

main();
