import { FastifyInstance } from 'fastify';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { env } from './env.js';
import sanitizeHtml from 'sanitize-html';

export async function routes(fastify: FastifyInstance) {
    fastify.get("/front.js", (req, res) => {
        const frontPath = join(import.meta.dirname, env.TRANS_FRONT_PATH, "front.js");
        const file = readFileSync(frontPath, 'utf8');
        res.type('text/js').send(file);
        return;
    })
    // ROAD OF TAG
    fastify.get('/part/about', function (req, reply) {
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, "about.html");
        const tag = readFileSync(frontPath, 'utf8');
        reply.type('text/html').send(tag);
    })
    fastify.get('/part/login*', function (req, reply) {
        let userName = (req.query as { name?: string }).name || "User"; // Get the name from query params
        userName = sanitizeHtml(userName); // Protection XSS Attacks
        if (!userName)
            userName = "User";
        // const token = jwt.sign({ userName }, env.SECRET_KEY, { expiresIn: '1h' }); // Définir le cookie avec le token
        const frontPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH || "", "login.html");
        let htmlContent = readFileSync(frontPath, 'utf8');
        htmlContent = htmlContent.replace('<span id="user-name"></span>', userName); // Insert Username
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
    // FAV ICON
    fastify.get('/favicon/favicon.ico', function (req, reply) {
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ICO_PATH, "favicon.ico");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/favicon/login.png', function (req, reply) {
        console.log("\n\nLOGIN\n\n");
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ICO_PATH, "login.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    fastify.get('/favicon/logout.png', function (req, reply) {
        console.log("\n\nLOGOUT\n\n");
        try {
            const frontPath = join(import.meta.dirname, env.TRANS_ICO_PATH, "logout.png");
            const tag = readFileSync(frontPath);
            reply.type('img/ico').send(tag)
        } catch (error) {
            reply.code(404).send("Fichier non trouvé");
        }
    });
    //CSS output
    fastify.get("/tail/output.css", (req, res) => {
        const frontPath = join(import.meta.dirname, env.TRANS_TAIL_PATH, "output.css");
        console.log(frontPath);
        const file = readFileSync(frontPath, 'utf8');
        res.type('text/css').send(file);
        return;
    })
}