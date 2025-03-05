import fastify from "fastify";
import httpProxy from '@fastify/http-proxy';
import cors from "@fastify/cors";

const app = fastify();

app.register(cors, {
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE"]
})

app.register(httpProxy, {
    upstream: 'http://user-management:8001',
    prefix: '/user-management',
});

app.listen({port: 8000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`);
});