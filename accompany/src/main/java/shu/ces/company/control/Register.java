package shu.ces.company.control;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.util.ResourceUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import shu.ces.company.model.User;
import shu.ces.company.service.RegisterService;
import java.io.*;

@Controller
public class Register {

    @GetMapping(value = "/register")
    public String register(){
        return "register";
    }

    @Autowired
    private RegisterService registerService;



    //  result >0 表示正常执行  result=0 表示出现错误
    @PostMapping(value="/register_add")
    @ResponseBody

    public String handleRegister(User user,@RequestParam("file") MultipartFile file){
        int result=1;
        String  headPortraitPath="";
        try{
            File path = new File(ResourceUtils.getURL("classpath:").getPath());
            File upload = new File(path.getAbsolutePath(),"static/upload/images/"+user.getPhone_number());

            if(!upload.exists()) upload.mkdirs();
            System.out.println("upload url:"+upload.getPath());
            String fileName=upload.getPath()+"/"+file.getOriginalFilename();
            int index=fileName.indexOf("upload");
            headPortraitPath=fileName.substring(index);
            System.out.println(headPortraitPath);
            BufferedOutputStream out = new BufferedOutputStream(
                    new FileOutputStream(new File(fileName)));
            out.write(file.getBytes());
            out.flush();
            out.close();

        }catch (IOException e) {
            e.printStackTrace();
            result=0;
        }

        user.setHead_portrait_path(headPortraitPath);
        result=registerService.add(user);
        System.out.println(result);

        return String.valueOf(result);
    }









}
