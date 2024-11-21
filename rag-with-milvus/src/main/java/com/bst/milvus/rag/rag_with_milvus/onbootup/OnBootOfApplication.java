package com.bst.milvus.rag.rag_with_milvus.onbootup;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.event.ContextRefreshedEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.dataformat.csv.CsvMapper;
import com.fasterxml.jackson.dataformat.csv.CsvSchema;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.theokanning.openai.embedding.Embedding;
import com.theokanning.openai.embedding.EmbeddingRequest;
import com.theokanning.openai.service.OpenAiService;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.HasCollectionReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.response.InsertResp;

@Component
public class OnBootOfApplication {

    @Value("${uri}")
    public String uri;

    @Value("${token}")
    public String token;

    @Value("${filePath}")
    public String filePath;

    @Value("${limitEmbeddingsTo}")
    public Long limitEmbeddingsTo;

    private MilvusClientV2 client;
    private OpenAiService openAiService;

    private boolean setUpNeeded = Boolean.TRUE;

    private static final String COLLECTION_NME = "commodity_collection_data";
    private static final Gson GSON_INSTANCE = new Gson();

    @EventListener
    public void setUpMivusCollection(ContextRefreshedEvent contextRefreshedEvent) {
        client = getMilvusClient();

        if (client.hasCollection(HasCollectionReq.builder().collectionName(COLLECTION_NME).build())) {
            setUpNeeded = Boolean.FALSE;
            System.out.println("**** Setting up EXISTS in Milvus DB ****");
        } else {
            System.out.println("**** Setting up in Milvus DB ****");
        }

        openAiService = new OpenAiService(token);
        createCollection(client);

        try {
            insertData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private MilvusClientV2 getMilvusClient() {
        ConnectConfig connectConfig = ConnectConfig.builder()
                .uri(uri)
                .build();

        return new MilvusClientV2(connectConfig);
    }

    private void createCollection(MilvusClientV2 client2) {
        CreateCollectionReq quickSetupReq = CreateCollectionReq.builder()
                .collectionName(COLLECTION_NME)
                .dimension(1536)
                .metricType("IP")
                .build();

        if (setUpNeeded) {
            System.out.println("**** Setting up Collection in Milvus DB ****");
            client.createCollection(quickSetupReq);
        }

    }

    private void insertData() throws IOException {

        if (setUpNeeded) {

            System.out.println("**** Inserting up records in Collection ****");

            CsvMapper csvMapper = new CsvMapper();

            File csvFile = new File(filePath);
            CsvSchema csvSchema = CsvSchema.builder().setUseHeader(true).build();
            Iterator<CsvDataObject> iterator = csvMapper.readerFor(CsvDataObject.class).with(csvSchema)
                    .readValues(csvFile);

            List<JSONObject> data = new ArrayList<>();
            Long count = 0l;

            while (iterator.hasNext()) {
                count++;
                CsvDataObject dataObject = iterator.next();
                JSONObject row = new JSONObject();

                row.put("vector", getEmbeddings(dataObject.toString(), count));
                row.put("id", count);
                row.put("state", dataObject.getState());
                row.put("district", dataObject.getDistrict());
                row.put("market", dataObject.getMarket());
                row.put("commodity", dataObject.getCommodity());
                row.put("variety", dataObject.getVariety());
                row.put("arrival_date", dataObject.getArrivalDate());
                row.put("min_price", dataObject.getMinPrice());
                row.put("max_price", dataObject.getMaxPrice());
                row.put("modal_price", dataObject.getModalPrice());

                data.add(row);

                if(count.equals(limitEmbeddingsTo))
                break;
            }

            InsertReq insertReq = InsertReq.builder()
                    .collectionName(COLLECTION_NME)
                    .data(data)
                    .build();

            System.out.println("**** Inserting ****");

            InsertResp insertResp = client.insert(insertReq);

            System.out.println(JSONObject.toJSON(insertResp));
        }

    }

    private static class CsvDataObject {
        @JsonProperty
        private String vector;
        @JsonProperty
        private String commodity;
        @JsonProperty
        private String state;
        @JsonProperty
        private String district;

        @Override
        public String toString() {
            return "commodity=" + commodity + ", state=" + state
                    + ", district=" + district + ", market=" + market + ", variety=" + variety
                    + ", arrival_date=" + arrival_date + ", min_price=" + min_price + ", max_price="
                    + max_price + ", modal_price=" + modal_price;
        }

        @JsonProperty
        private String market;
        @JsonProperty
        private String variety;
        @JsonProperty
        private String arrival_date;
        @JsonProperty
        private String min_price;
        @JsonProperty
        private String max_price;
        @JsonProperty
        private String modal_price;

        public String getVector() {
            return vector;
        }

        public String getCommodity() {
            return commodity;
        }

        public String getState() {
            return state;
        }

        public String getDistrict() {
            return district;
        }

        public String getMarket() {
            return market;
        }

        public String getVariety() {
            return variety;
        }

        public String getArrivalDate() {
            return arrival_date;
        }

        public String getMinPrice() {
            return min_price;
        }

        public String getMaxPrice() {
            return max_price;
        }

        public String getModalPrice() {
            return modal_price;
        }

        public List<Float> toFloatArray() {
            return GSON_INSTANCE.fromJson(vector, new TypeToken<List<Float>>() {
            }.getType());
        }
    }

    private List<Float> getEmbeddings(String faq_line, Long count) {
        EmbeddingRequest embeddingRequest = EmbeddingRequest.builder()
                .model("text-embedding-3-small")
                .input(faq_line)
                .build();

        List<Embedding> embeddings = openAiService.createEmbeddings(embeddingRequest).getData();

        String sliced_embeddings = embeddings.get(0).getEmbedding().toString();
        StringTokenizer dimensions = new StringTokenizer(
                sliced_embeddings.substring(1, sliced_embeddings.length() - 1), ",");

        List<Float> embeddings_flt = new ArrayList<>();
        while (dimensions.hasMoreTokens()) {
            embeddings_flt.add(new Float(dimensions.nextElement().toString()));
        }
        if (count % 10 == 0)
            System.out.println(count + " Embeddings created ****");

        return embeddings_flt;
    }
}
