import { FastifyInstance } from 'fastify';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { env } from './env.js';
import {Client_db, CreateClient} from "./database.js";

export async function routes(fastify: FastifyInstance) {
    fastify.get("/front.js", (req, res) => {
        const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
        const file = readFileSync(frontPath, 'utf8');
        res.type('text/js').send(file);
        return;
    })
    // ROAD OF BALISE
    fastify.get('/part/about', function (req, reply) {
        console.log("=======about.html===========")
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "about.html");
        const balise = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(balise);
    })
    fastify.get('/part/login', function (req, reply) {
        console.log("=======login.html===========")
        // fastify.post("/register", CreateClient);
        const frontPath = join(import.meta.dirname,  env.TRANS_VIEWS_PATH, "login.html");
        const balise = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(balise)
    })
    fastify.get('/part/register', function (req, reply) {
        console.log("=======register.html===========")
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "register.html");
        const balise = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(balise)
    })
    fastify.get('/part/contact', function (req, reply) {
        console.log("=======contact.html===========")
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "contact.html");
        const balise = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(balise)
    })
    // FAV ICON
    fastify.get('/favicon.ico', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ICO_PATH, "favicon.ico");
            const balise = readFileSync(frontPath);
            reply.type('img/ico').send(balise)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
}