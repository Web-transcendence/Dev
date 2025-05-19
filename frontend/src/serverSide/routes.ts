import { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import { readFileSync } from 'node:fs';
import { join ,extname } from 'node:path';
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
    const htmlRoutes = [
        "about",
        "toKnow",
        "logout",
        "login",
        "register",
        "connect",
        "contact",
        "profile",
        "home",
        "towerRemote",
        "towerMode",
        "towerWatch",
        "pongRemote",
        "pongLocal",
        "pongWatch",
        "pongMode",
        "tournaments",
        "lobby",
        "factor",
        "matchHistory",
        "brackets",
    ];

    htmlRoutes.forEach(route => {
        fastify.get(`/part/${route}`, (req, reply) => {
            let filename = route === "logout" ? "home.html" : `${route}.html`;
            try {
                const fullPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, filename);
                const html = readFileSync(fullPath, "utf8");
                reply.type("text/html").send(html);
            } catch (error) {
                reply.code(404).send(`Page ${filename} non trouvée`);
            }
        });
    });

        // Route par défaut : redirige toute autre page sur index.html
        fastify.get('/part/:name', (req, reply) => {
            try {
                const fullPath = join(import.meta.dirname, env.TRANS_VIEWS_PATH, 'index.html');
                const html = readFileSync(fullPath, 'utf8');
                reply.type('text/html').send(html);
            } catch (error) {
                reply.code(500).send('Erreur serveur : index.html introuvable');
            }
        });

    // FAV ICON

    const allowedImages = new Set([
        'favicon.png',
        'favicon.ico',
        'fourRobots.png',
        'sixteenRobots.png',
        'login.png',
        'logout.png',
        'BigLock.png',
        'solo.png',
        'vsia.png',
        'remote.png'
    ]);
     fastify.get('/images/:imageName', async (req: FastifyRequest<{ Params: { imageName: string } }>, reply: FastifyReply) => {
         const { imageName } = req.params;

         if (!allowedImages.has(imageName)) {
             return reply.code(404).send("Image non autorisée");
         }

         try {
             const filePath = join(import.meta.dirname, env.TRANS_IMG_PATH, imageName);
             const fileData = readFileSync(filePath);
             reply.type('image/png').send(fileData);
         } catch {
             reply.code(404).send("Fichier non trouvé");
         }
     });
     //Pong assets
    // const pongImages = new Set([
    //     'ballup.png',
    //     'bardown.png',
    //     'barup.png',
    //     'pong.png'
    // ]);
}