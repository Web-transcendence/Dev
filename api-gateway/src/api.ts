import Fastify, {FastifyReply, FastifyRequest} from "fastify";
import httpProxy from '@fastify/http-proxy';
import cors from "@fastify/cors";
import jwt, {JwtPayload} from 'jsonwebtoken';


const SECRET_KEY = /*process.env.SECRET_KEY || */ "secret_key";

const app = Fastify();


app.register(cors, {
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE"]
})

async function authentificate (req: FastifyRequest, reply: FastifyReply) {
    if (req.url === "/user-management/sign-up" || req.url === "/user-management/sign-in")
        return ;
    try {
        const authHeader = req.headers.authorization;
        if (!authHeader) {
            return reply.status(401).send({ error: "Unauthorized - No token provided" });
        }

        const token = authHeader.split(" ")[1];
        if (!token) {
            return reply.status(401).send({ error: "Unauthorized - No token provided" });
        }

        const decoded = jwt.verify(token, SECRET_KEY) as JwtPayload;
        req.headers.name = decoded.name;
    }
    catch (error) {
        return reply.status(401).send({ error: "Unauthorized - invalid token" });
    }
}


app.register(httpProxy, {
    upstream: 'http://user-management:5000',
    prefix: '/user-management',
    http2: false,
    preHandler: authentificate
});

app.listen({port: 3000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`);
});