import fastify from "fastify";
import httpProxy from '@fastify/http-proxy';
import cors from "@fastify/cors";

const app = fastify();

app.register(cors, {
    origin: "*",
    method: ["GET", "POST", "PUT", "DELETE"]
})

app.register(httpProxy, {
    upstream: 'http://localhost:8000',
    prefix: '/user-management',
    rewritePrefix: '/user-management'
});

app.listen({port: 8001}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`);
});

