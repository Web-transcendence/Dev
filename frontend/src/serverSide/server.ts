import Fastify, {FastifyReply, FastifyRequest} from "fastify";
import httpProxy from '@fastify/http-proxy';
// import cors from "@fastify/cors";
import jwt, {JwtPayload} from 'jsonwebtoken';
import {readFileSync} from "node:fs";
import {join} from "node:path";
import fastifyStatic from "@fastify/static";
import {env} from "./env";
import {routes} from "./routes";

// process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
//
const httpsOptions = {
    https: {
        key: readFileSync(join(import.meta.dirname, '../../secure/key.pem')),      // Private key
        cert: readFileSync(join(import.meta.dirname, '../../secure/cert.pem'))     // Certificate
    },
};


const SECRET_KEY = /*process.env.SECRET_KEY || */ "secret_key";

const app = Fastify(httpsOptions);


// app.register(cors, {
//     origin: "*",
//     methods: ["GET", "POST", "PUT", "DELETE"]
// })



async function authentificate (req: FastifyRequest, reply: FastifyReply) {
    if (req.url === "/user-management/login" || req.url === "/user-management/register" || req.url === "/user-management/auth/google" || req.url === "/user-management/2faVerify")
        return;
    try {
        const authHeader = req.headers.authorization;
        if (!authHeader)
            return reply.status(401).send({ error: "Unauthorized - No token provided" });

        const token = authHeader.split(" ")[1];
        if (!token)
            return reply.status(401).send({ error: "Unauthorized - No token provided" });

        const decoded = jwt.verify(token, SECRET_KEY) as JwtPayload;
        req.headers.id = decoded.id;
    }
    catch (error) {
        return reply.status(401).send({ error: "Unauthorized - invalid token" });
    }
}

app.get('/authJWT', (req: FastifyRequest, res: FastifyReply) => {
    authentificate(req, res);
    if (!req.headers.id)
        return res.status(401).send({ message: "Unauthorized - No token provided" });
    return res.status(200).send({message: "Authentication successfull"});
})

app.register(fastifyStatic, {
    root: join(import.meta.dirname, "..", "..", "public"),
    prefix: "/static/"
})

app.register(routes)

app.register(httpProxy, {
    upstream: 'http://match-server:4443',
    prefix: '/match-server',
    websocket: true,
});

app.register(httpProxy, {
    upstream: 'http://user-management:5000',
    prefix: '/user-management',
    http2: false,
    preHandler: authentificate
});

app.register(httpProxy, {
    upstream: 'http://social:6500',
    prefix: '/social',
    http2: false,
    preHandler: authentificate
});

app.register(httpProxy, {
    upstream: 'http://tournament:7000',
    prefix: '/tournament',
    http2: false,
    preHandler: authentificate
});

app.register(async function (instance) {
    instance.register(fastifyStatic, {
        root: join(import.meta.dirname, env.TRANS_ASSETS_PATH),
        prefix: "/tower-defense/",
        decorateReply: false, // 👈 empêche le conflit de sendFile
    });
});

app.get("/*", (req, res) => { // Route pour la page d'accueil
    const pagePath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "index.html");
    const readFile = readFileSync(pagePath, 'utf8');
    res.status(202).type('text/html').send(readFile);
});


app.listen({port: 4000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}, ${join(import.meta.dirname, '../secure/cert.pem')}`);
});