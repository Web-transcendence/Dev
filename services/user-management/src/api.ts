import fastify from 'fastify'
import userRoutes from "./routes.js"
import { FastifySSEPlugin } from "fastify-sse-v2";
import { FastifyReply } from "fastify";

const app = fastify();

app.register(FastifySSEPlugin);
app.register(userRoutes);

export const connectedUsers = new Map<string, FastifyReply>();

app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})