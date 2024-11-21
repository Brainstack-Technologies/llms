package com.bst.milvus.rag.rag_with_milvus.service;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import com.theokanning.openai.completion.chat.ChatCompletionRequest;
import com.theokanning.openai.completion.chat.ChatCompletionResult;
import com.theokanning.openai.completion.chat.ChatMessage;
import com.theokanning.openai.completion.chat.SystemMessage;
import com.theokanning.openai.completion.chat.UserMessage;
import com.theokanning.openai.embedding.Embedding;
import com.theokanning.openai.embedding.EmbeddingRequest;
import com.theokanning.openai.service.OpenAiService;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.response.SearchResp;

@Component
public class MilvusCommoditySrchWithRAG {

    @Value("${uri}")
    public String uri;

    @Value("${token}")
    public String token;

    private static final String COLLECTION_NME = "commodity_collection_data";
    private static final String SYSTEM_PROMPT = "\"\"\"\n" + //
            "Human: You are a smart and knowledgeable AI assistant. Your name is BSTBot. You help users discover commodities and get recommendations based on their commodities prices. You can also answer questions about specific prices, commodities, market, district and state.\n"
            + //
            "\"\"\"";

    private static final String USER_PROMPT = "f\"\"\"\n" + //
            "Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.\n"
            + //
            "<context>\n" + //
            "vrContext" + //
            "</context>\n" + //
            "<question>\n" + //
            "vector_Question" + //
            "</question>\n" + //
            "\"\"\"";

    private MilvusClientV2 client;
    private OpenAiService openAiService;
    private SearchResp searchResp;

    public String milvusCommoditySrchWithRAG(String vector_Question) {
        searchData(vector_Question);
        return generteLlmBasedResponse(vector_Question);
    }

    private void searchData(String vector_Question) {
        List<List<Float>> query_vectors = Arrays.asList(getEmbeddings(vector_Question));

        if (client == null)
            client = getMilvusClient();

        SearchReq searchReq = SearchReq.builder()
                .collectionName(COLLECTION_NME)
                .data(query_vectors)
                .outputFields(Arrays.asList("commodity", "modal_price", "market", "district", "state"))
                .topK(10) // The number of results to return
                .build();

        searchResp = client.search(searchReq);

        System.out.println(" ** searchResp **" + searchResp.getSearchResults().get(0).toString());
    }

    private String generteLlmBasedResponse(String vector_Question) {

        System.out.println(" *** In generteLlmBasedResponse Func ***");

        String context = "\n".concat(searchResp.getSearchResults().get(0).toString());

        List<ChatMessage> chatmessages = new ArrayList<>();

        chatmessages.add(new SystemMessage(SYSTEM_PROMPT));
        chatmessages.add(
                new UserMessage(USER_PROMPT.replace("vrContext", context).replace("vector_Question", vector_Question)));

        ChatCompletionResult result = openAiService.createChatCompletion(ChatCompletionRequest.builder()
                .model("gpt-4o-mini")
                .messages(chatmessages)
                .build());

        System.out.println(
                " *** LLM generted result ***" + result.getChoices().get(0).getMessage().getContent());

        return result.getChoices().get(0).getMessage().getContent();
    }

    private MilvusClientV2 getMilvusClient() {
        ConnectConfig connectConfig = ConnectConfig.builder()
                .uri(uri)
                .build();

        return new MilvusClientV2(connectConfig);
    }

    private List<Float> getEmbeddings(String faq_line) {
        EmbeddingRequest embeddingRequest = EmbeddingRequest.builder()
                .model("text-embedding-3-small")
                .input(faq_line)
                .build();

        if (null == openAiService)
            openAiService = new OpenAiService(token);

        List<Embedding> embeddings = openAiService.createEmbeddings(embeddingRequest).getData();

        String sliced_embeddings = embeddings.get(0).getEmbedding().toString();
        StringTokenizer dimensions = new StringTokenizer(
                sliced_embeddings.substring(1, sliced_embeddings.length() - 1), ",");

        List<Float> embeddings_flt = new ArrayList<>();
        while (dimensions.hasMoreTokens()) {
            embeddings_flt.add(new Float(dimensions.nextElement().toString()));
        }
        return embeddings_flt;
    }

}
