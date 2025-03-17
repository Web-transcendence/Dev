import * as Schema from "./schema.js";
import sanitizeHtml from "sanitize-html";
import {User} from "./User.js";
import {FastifyReply, FastifyRequest, FastifyInstance } from "fastify";


export default async function userRoutes(app: FastifyInstance) {

    app.post('/sign-up', async (req, res) => {
        try {
            console.log("sign-up");
            const zod_result = Schema.signUpSchema.safeParse(req.body);
            if (!zod_result.success)
                return res.status(400).send({json: zod_result.error.format()});
            let {nickName, email, password} = {nickName: sanitizeHtml(zod_result.data.nickName), email: sanitizeHtml(zod_result.data.email), password: sanitizeHtml(zod_result.data.password)};
            if (!nickName || !email || !password)
                return res.status(454).send({error: "All information are required !"});

            const addRes = await User.addClient(nickName, email, password);
            if (!addRes.success) {
                return res.status(409).send({error: `${addRes.result} already exists`});
            }
            return res.status(201).send({token: addRes.result, redirect: "post/login"});
        }
        catch(err) {
            return res.status(500).send({error: "Server error: ", err});
        }
    });

    app.get('/getProfile', async (req, res) => {

        try {
            const zod_result = Schema.profileSchema.safeParse(req.headers);
            if (!zod_result.success)
                return res.status(400).send({json: zod_result.error.format()});
            let name = sanitizeHtml(zod_result.data.name);
            if (!name)
                return res.status(454).send({error: "All information are required !"});

            const user = new User(name);
            const profileData = user.getProfile();
            return res.status(200).send(profileData);
        } catch (err) {
            return res.status(500).send({error: "Server error: ", err});
        }


    })

    app.post('/sign-in', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            console.log("sign-in");
            const zod_result = Schema.signInSchema.safeParse(req.body);
            if (!zod_result.success)
                return res.status(400).send({json: zod_result.error.format()});
            let {name, password} = {
                name: sanitizeHtml(zod_result.data.name),
                password: sanitizeHtml(zod_result.data.password)
            };
            if (!name || !password)
                return res.status(454).send({error: "All information are required !"});

            const user = new User(name);
            if (user.getStatus() === "NotExists")
                return res.status(401).send({error: "Invalid username"});

            if (await user.isPasswordValid(password))
                return res.status(201).send({token: user.makeToken(), username: name , redirect: "post/login"});
            else
                return res.status(401).send({error: "Invalid password"});
        }
        catch (err) {
            res.status(500).send({error: "Server error: ", err});
        }
    });
}