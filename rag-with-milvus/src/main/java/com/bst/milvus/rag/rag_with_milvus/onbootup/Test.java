package com.bst.milvus.rag.rag_with_milvus.onbootup;

public class Test {
    public static void main(String[] args) {
        for (Long count=0l; count<100; count++) {
            System.out.println(count + "   :   " +count % 10);
            if (count % 10 == 0)
            System.out.println(count + "   :   " + count % 10 + "  Embeddings created ");
        }
    }
}
