package com.bst.milvus.rag.rag_with_milvus.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.bst.milvus.rag.rag_with_milvus.service.MilvusCommoditySrchWithRAG;

import lombok.RequiredArgsConstructor;

@RestController
@RequiredArgsConstructor
public class ClientRagRequestResponse {

    @Autowired
    MilvusCommoditySrchWithRAG milvusCommoditySrchWithRAG;

    @RequestMapping("/ask")
    public ResponseEntity<?> respondToAsk(@RequestParam String vector_Question) {
        System.out.println("****  Received : " + vector_Question + "   ****");
        if(!vector_Question.isEmpty() && null != vector_Question) {
            return ResponseEntity.ok(milvusCommoditySrchWithRAG.milvusCommoditySrchWithRAG(vector_Question));
        }
        return ResponseEntity.badRequest().build();
    }
    
}
