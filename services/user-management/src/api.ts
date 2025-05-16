import fastify from 'fastify'
import userRoutes from "./routes.js"
import {googleAuth} from "./googleApi.js";
import { FastifySSEPlugin } from "fastify-sse-v2";
import { FastifyReply } from "fastify";
import fastifyCookie from '@fastify/cookie';
import fastifyJwt from '@fastify/jwt';

export const app = fastify();

export const INTERNAL_PASSWORD = process.env.SECRET_KEY;

app.register(fastifyCookie);
app.register(fastifyJwt, {
    secret: INTERNAL_PASSWORD,
    cookie: {
        cookieName: 'token',
        signed: false
    }
});

app.register(FastifySSEPlugin);
app.register(userRoutes);
app.post('/auth/google', googleAuth);

export const connectedUsers = new Map<number, FastifyReply>();



app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})
