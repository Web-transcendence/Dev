import Fastify from "fastify";
import httpProxy from '@fastify/http-proxy';
import cors from "@fastify/cors";
// import {readFileSync} from "node:fs";
// import {join} from "node:path";

// const httpsOptions = {
//     https: {
//         key: readFileSync(join(import.meta.dirname, '../secure/key.pem')),      // Private key
//         cert: readFileSync(join(import.meta.dirname, '../secure/cert.pem'))     // Certificate
//     },
//     logger: true
// };
//
// const app = Fastify(httpsOptions)
// fastify.register(fastifyCors, {
//     origin: ['http://localhost:5000', 'http://localhost:5000'], // Allowed origins
//     methods: ['GET', 'POST', 'PUT', 'DELETE'], // Allowed methods
//     allowedHeaders: ['Content-Type', 'Authorization'], // Allowed headers
//     credentials: true, // Enable cookies and authorization headers
// });

const app = Fastify();

app.register(cors, {
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE"]
})

app.register(httpProxy, {
    upstream: 'http://user-management:5000',
    prefix: '/user-management',
    http2: false
});

app.listen({port: 3000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`);
});