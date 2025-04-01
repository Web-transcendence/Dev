import { FastifyInstance } from 'fastify';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { env } from './env.js';

export async function routes(fastify: FastifyInstance) {
    fastify.get("/front.js", (req, res) => {
        const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
        const file = readFileSync(frontPath, 'utf8');
        res.type('text/js').send(file);
        return;
    })
    fastify.get("/pong.js", (req, res) => {
        const pongPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "pong.js");
        const file = readFileSync(pongPath, 'utf8');
        res.type('text/js').send(file);
        return;
    })
    // ROAD OF TAG
    fastify.get('/part/about', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "about.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    fastify.get('/part/connected', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH || "", "connected.html");
        let htmlContent = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(htmlContent);
    })
    fastify.get('/part/logout', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH || "", "home.html");
        let htmlContent = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(htmlContent);
    })
    fastify.get('/part/login', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH || "", "login.html");
        let htmlContent = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(htmlContent);
    })
    fastify.get('/part/register', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "register.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    fastify.get('/part/contact', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "contact.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    fastify.get('/part/profile', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "profile.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag)
    })
    // FAV ICON
    fastify.get('/favicon.ico', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ICO_PATH, "favicon.ico");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/login.png', function (req, reply) {
        console.log("\n\nLOGIN\n\n");
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ICO_PATH, "login.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/logout.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ICO_PATH, "logout.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/part/factor', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "factor.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    fastify.get('/part/pong', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "pong.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    // Png For Pong
    fastify.get('/assets/ballup.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "ballup.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier ballup non trouvé");
        }
    });
    fastify.get('/assets/bardown.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "bardown.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier bardown non trouvé");
        }
    });
    fastify.get('/assets/barup.png', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ASSETS_PATH, "barup.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier barup non trouvé");
        }
    });
}