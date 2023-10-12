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
    "Find and extract a common writing structure that repeats throughout the book from your context.";
  const queryFieldGuide = await fieldGuideChatEngine.chat(FIELD_GUIDE_QUERY, [
    {
      content:
        "You are an AI assistant that helps users find a common writing structure that repeats throughout books. In your context there's the 'Birds of the British Empire' book from W.T. Greene. In this field guide from the 19th-century the author describes birds found on the british empire.",
      role: "system",
    },
  ]);
  const fieldGuideDescription = queryFieldGuide.toString();
  console.log("field guide description:", fieldGuideDescription);

  const fictionAuthorIndex = await getVectorIndex("fiction-author");
  const fictionAuthorChatEngine = await getChatEngine(fictionAuthorIndex);
  const FICTION_AUTHOR_QUERY = `Based on the following structure, create a new bird description that extracts the essence of the book 'Jane Eyre', from 19th-century author Charlotte Bronte.
  '''
  ${fieldGuideDescription}
  '''
  `;
  const queryFictionAuthor = await fictionAuthorChatEngine.chat(
    FICTION_AUTHOR_QUERY,
    [
      {
        content:
          "Answer questions SOLELY based on the context provided to you. DO NOT use your previous knowledge.",
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

async function describeJaneEyre() {
  const llm = new LlamaOpenAI({ model: "gpt-3.5-turbo", temperature: 0.0 });

  const CHARACTER_BIRD_PROMPT = `You are an AI assistant who describes characters from novels as if they were birds.

  To do that, you use the book "Birds from the British Empire" by W T Greene as a reference.
  
  Notice that this book uses the following structure to describe a bird:
  
  Paragraph 1: Physical description of the bird. Color, patterns, body parts, length. Any difference between genders. Etc.
  
  Paragraph 2: Geography. Where it is found. Type of nest/habitat. Diet.
  
  Paragraph 3: Breeding, broods, mating. History.
  
  Paragraph 4: Personality. Behaviour. Politics. With Examples.
  
  Paragraph 5: Colonialism. Captivity. Opinion/bias towards the bird. Relationship to "people" (empire).
  
  You should use this structure to describe the birds. You don't need to name the paragraph, just follow this structure.`;

  const characterAsABird = await llm.chat([
    { role: "system", content: CHARACTER_BIRD_PROMPT },
    {
      role: "user",
      content:
        "Imagine if the character Bertha Mason from Charlotte Bronte's novel Jane Eyre were a bird in the Birds from the British Empire book. Describe this bird.",
    },
  ]);

  console.log(">>> CHARACTER AS A BIRD:", characterAsABird.message.content);

  const imagePrompt = await llm.chat([
    {
      content: `You are an assistant that receives a bird description and formats this description as a query to Open AI's DALL-E image generator prompt. Your response needs to have a max of 1000 characters.`,
      role: "system",
    },
    { content: characterAsABird.message.content, role: "user" },
  ]);

  console.log(">>> IMAGE PROMPT:", imagePrompt);

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  const imageResponse = await openai.images.generate({
    prompt: imagePrompt.message.content,
    n: 3,
    size: "1024x1024",
  });

  console.log(imageResponse);
}

describeJaneEyre();
