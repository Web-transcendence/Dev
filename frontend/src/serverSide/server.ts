import Fastify, {FastifyReply, FastifyRequest} from "fastify";
import httpProxy from '@fastify/http-proxy';
import jwt, {JwtPayload} from 'jsonwebtoken';
import {readFileSync} from "node:fs";
import {join} from "node:path";
import fastifyStatic from "@fastify/static";
import {env} from "./env";
import {routes} from "./routes";
import { createClient } from "@supabase/supabase-js";
import { z } from "zod";
import rateLimit from "@fastify/rate-limit";

const httpsOptions = {
    https: {
        key: readFileSync(join(import.meta.dirname, '../../secure/key.pem')),      // Private key
        cert: readFileSync(join(import.meta.dirname, '../../secure/cert.pem'))     // Certificate
    },
};

const SECRET_KEY = process.env.SECRET_KEY;

const app = Fastify(httpsOptions);
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// GotoRegister
await app.register(rateLimit, {
    global: false,
});

const contactSchema = z.object({
    email: z.string().email(),
    type: z.string().min(1, "Le type ne peut pas être vide"),
});

app.post("/api/add-contact",{config: {rateLimit: {max: 5, timeWindow: "1 minute",},},}, async (request, reply) => {
    const validation = contactSchema.safeParse(request.body);
    if (!validation.success) {
        return reply.status(400).send({ error: validation.error.errors[0].message });
    }

    const { email, type} = validation.data;

    const { error } = await supabase.from("contacts").insert([{ email, type }]);

    if (error) {
        if (error.code === "23505") {
            return reply.status(409).send({ error: "Cet email est déjà enregistré." });
        }
        return reply.status(500).send({ error: error.message });
    }

    return reply.send({ success: true });
});

const messageSchema = z.object({
    email: z.string().email(),
    message: z.string().min(1, "Le message ne peut pas être vide"),
});

app.post("/api/get-message", {config: {rateLimit: {max: 5, timeWindow: "1 minute",},},}, async (request, reply) => {
        const validation = messageSchema.safeParse(request.body);
        if (!validation.success) {
            return reply.status(400).send({ error: validation.error.errors[0].message });
        }

        const { email, message} = validation.data;
        const { error } = await supabase.from("messagerie").insert([{ email, message }]);

        if (error) return reply.status(500).send({ error: error.message });
        return reply.send({ success: true });
    });

// Transcendance
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
    http2: false,
    preHandler: authentificate
});

app.register(httpProxy, {
    upstream: 'http://tower-defense:2246',
    prefix: '/tower-defense',
    http2: false,
    preHandler: authentificate
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

app.register(httpProxy, {
    upstream: 'http://tower-defense:2246/ws',
    prefix: '/tower-defense/ws',
    websocket: true
});

app.register(httpProxy, {
    upstream: 'http://match-server:4443/ws',
    prefix: '/match-server/ws',
    websocket: true
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