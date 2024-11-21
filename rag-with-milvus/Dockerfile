#Stage Build
FROM maven:3.8.3-openjdk-17 AS build
COPY src /home/app/src
COPY pom.xml /home/app

RUN mvn -f /home/app/pom.xml clean package -DskipTests

#Package
FROM openjdk:17-jdk-alpine as light-jdk
RUN apk add --no-cache binutils

RUN jlink \
         --add-modules ALL-MODULE-PATH \
         --strip-debug \
         --no-man-pages \
         --no-header-files \
         --compress=2 \
         --output /jre

FROM alpine:latest
RUN apk add -U tzdata
RUN apk add --no-cache curl
ENV TZ=Asia/Kolkata
RUN cp /usr/share/zoneinfo/Asia/Kolkata /etc/localtime

ENV JAVA_HOME=/jre
ENV PATH="${JAVA_HOME}/bin:${PATH}"
COPY --from=light-jdk /jre $JAVA_HOME
RUN mkdir -p /site/uploads
COPY --from=build /home/app/target/edmp-pwa.jar /usr/local/edmp-pwa.jar
EXPOSE 8080
ENTRYPOINT ["java","-jar","/usr/local/rag-with-milvus.jar"]