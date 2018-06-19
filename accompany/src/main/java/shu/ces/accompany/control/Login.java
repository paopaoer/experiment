package shu.ces.accompany.control;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;


@Controller
public class Login {

    @GetMapping(value = "/login")
    public String login(){

        return "login";
    }

}
